#include "bril/MLIR2JSON.h"
#include "bril/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
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
  json convertBlock(mlir::Block &block, std::string blockName, llvm::DenseMap<mlir::Block*, std::string> blockNames);
  json convertOperation(mlir::Operation &op, llvm::DenseMap<mlir::Block*, std::string> blockNames);

  llvm::DenseMap<mlir::Value, std::string> valueToName;
  
  
  int nextVarId = 0;
  
  std::string getValueName(mlir::Value value);
  
  std::string getTypeString(mlir::Type type);
  
  std::string currentBlockName;
  
  void handleBlockArguments(mlir::Block &block, json &instrs, std::string blockName);
  
  struct PhiNode {
    std::string dest;
    std::vector<std::string> args;
    std::vector<std::string> labels;
    std::string type;
  };
  
  llvm::DenseMap<mlir::Block*, std::vector<PhiNode>> blockPhiNodes;
  
  llvm::DenseMap<mlir::Block*, std::vector<mlir::Block*>> blockPredecessors;
  
  using BlockArgMap = llvm::DenseMap<mlir::Block*, std::string>;
  llvm::DenseMap<mlir::Value, BlockArgMap> phiValueSources;
};

std::string MLIRToJSONConverter::getValueName(mlir::Value value) {
  auto it = valueToName.find(value);
  if (it != valueToName.end()) {
    return it->second;
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

void MLIRToJSONConverter::handleBlockArguments(mlir::Block &block, json &instrs, std::string blockName) {
  for (auto arg : block.getArguments()) {
    auto it = phiValueSources.find(arg);
    if (it != phiValueSources.end()) {
      PhiNode phi;
      phi.dest = getValueName(arg);
      phi.type = getTypeString(arg.getType());
      
      for (auto &pred : it->second) {
        std::string predName = blockName;
        std::string argName = pred.second;
        
        phi.labels.push_back(predName);
        phi.args.push_back(argName);
      }
      
      json phiInstr;
      phiInstr["op"] = "phi";
      phiInstr["dest"] = phi.dest;
      phiInstr["type"] = phi.type;
      phiInstr["args"] = phi.args;
      phiInstr["labels"] = phi.labels;
      
      instrs.push_back(phiInstr);
    }
    else {
      std::string name = getValueName(arg);
      std::string type = getTypeString(arg.getType());
      
      json phiInstr;
      phiInstr["op"] = "phi";
      phiInstr["dest"] = name;
      phiInstr["type"] = type;
      phiInstr["args"] = json::array();
      phiInstr["labels"] = json::array();
      
      instrs.push_back(phiInstr);
    }
  }
}

json MLIRToJSONConverter::convertOperation(mlir::Operation &op, llvm::DenseMap<mlir::Block*, std::string> blockNames) {
  json instr;
  
  if (auto constOp = dyn_cast<mlir::arith::ConstantIntOp>(&op)) {
    instr["op"] = "const";
    instr["dest"] = getValueName(constOp->getResult(0));
    instr["type"] = getTypeString(constOp->getResult(0).getType());
    
          if (constOp->getResult(0).getType().isInteger(1)) {
        auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
        if (intAttr) {
          bool boolValue = intAttr.getInt() != 0;
          instr["value"] = boolValue ? "true" : "false";
        } else {
          instr["value"] = "false";
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
    instr["args"] = {getValueName(printOp.getValue())};
  }
  else if (auto nopOp = dyn_cast<mlir::bril::Nop>(&op)) {
    instr["op"] = "nop";
  }
  else if (auto brOp = dyn_cast<mlir::cf::BranchOp>(&op)) {
    instr["op"] = "jmp";
    // Get a unique identifier for the destination block
    mlir::Block *destBlock = brOp.getDest();
    // We'll replace this with actual labels in convertFunction
    // int blockIndex = std::distance(&destBlock->getParent()->front(), destBlock);
    instr["labels"] = { blockNames[destBlock] };

    
    if (brOp.getNumOperands() > 0) {
      mlir::Block *destBlock = brOp.getDest();
      for (unsigned i = 0; i < brOp.getNumOperands(); i++) {
        mlir::Value arg = brOp.getOperand(i);
        mlir::Value blockArg = destBlock->getArgument(i);
        std::string argName = getValueName(arg);
        
        phiValueSources[blockArg][op.getBlock()] = argName;
      }
    }
  }
  else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(&op)) {
    instr["op"] = "br";
    instr["args"] = {getValueName(condBrOp.getCondition())};
    
    // Get unique identifiers for destination blocks
    mlir::Block *trueBlock = condBrOp.getTrueDest();
    mlir::Block *falseBlock = condBrOp.getFalseDest();
    
    // Calculate block indices
    // int trueBlockIndex = std::distance(&trueBlock->getParent()->front(), trueBlock);
    // int falseBlockIndex = std::distance(&falseBlock->getParent()->front(), falseBlock);
    
    instr["labels"] = {
      blockNames[trueBlock],
      blockNames[falseBlock]
    }; 
    
    if (condBrOp.getNumTrueOperands() > 0) {
      mlir::Block *trueBlock = condBrOp.getTrueDest();
      for (unsigned i = 0; i < condBrOp.getNumTrueOperands(); i++) {
        mlir::Value arg = condBrOp.getTrueOperand(i);
        mlir::Value blockArg = trueBlock->getArgument(i);
        std::string argName = getValueName(arg);
        
        phiValueSources[blockArg][op.getBlock()] = argName;
      }
    }
    
    if (condBrOp.getNumFalseOperands() > 0) {
      mlir::Block *falseBlock = condBrOp.getFalseDest();
      for (unsigned i = 0; i < condBrOp.getNumFalseOperands(); i++) {
        mlir::Value arg = condBrOp.getFalseOperand(i);
        mlir::Value blockArg = falseBlock->getArgument(i);
        std::string argName = getValueName(arg);
        
        phiValueSources[blockArg][op.getBlock()] = argName;
      }
    }
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

json MLIRToJSONConverter::convertBlock(mlir::Block &block, std::string blockName, llvm::DenseMap<mlir::Block*, std::string> blockNames) {
  json instrs = json::array();
  
  if (blockName != "entry") {
    json labelInstr;
    labelInstr["label"] = blockName;
    instrs.push_back(labelInstr);
  }

  handleBlockArguments(block, instrs, blockName);
  
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
  
  llvm::DenseMap<mlir::Block*, std::string> blockNames;
  llvm::StringMap<int> nameCount;
  
  blockNames[&funcOp.getBody().front()] = "entry";
  
  int blockId = 0;
  for (auto &block : funcOp.getBody()) {
    if (&block != &funcOp.getBody().front()) {
      std::string name = "block" + std::to_string(blockId++);
      blockNames[&block] = name;
    }
  }
  
  for (auto &block : funcOp.getBody()) {
    for (auto &op : block) {
      if (auto brOp = dyn_cast<mlir::cf::BranchOp>(&op)) {
        mlir::Block *dest = brOp.getDest();
        blockPredecessors[dest].push_back(&block);
      }
      else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(&op)) {
        mlir::Block *trueDest = condBrOp.getTrueDest();
        mlir::Block *falseDest = condBrOp.getFalseDest();
        blockPredecessors[trueDest].push_back(&block);
        blockPredecessors[falseDest].push_back(&block);
      }
    }
  }
  
  json args = json::array();
  for (auto arg : funcOp.getArguments()) {
    json argObj;
    argObj["name"] = getValueName(arg);
    // std::cout << getTypeString(arg.getType()) << std::endl;
    argObj["type"] = getTypeString(arg.getType());
    args.push_back(argObj);
  }
  func["args"] = args;
  
  json allInstrs = json::array();
  for (auto &block : funcOp.getBody()) {
    std::string blockName = blockNames[&block];
    currentBlockName = blockName;
    
    json blockInstrs = convertBlock(block, blockName, blockNames);
    for (auto &instr : blockInstrs) {
      allInstrs.push_back(instr);
    }
  }

  mlir::FunctionType fnType = funcOp.getFunctionType();
  if (fnType.getNumResults() > 0) {
    func["type"] = getTypeString(fnType.getResult(0));
  }
  
  // for (auto &instr : allInstrs) {
  //   if (instr.contains("op") && (instr["op"] == "jmp" || instr["op"] == "br") && instr.contains("labels")) {
  //     json newLabels = json::array();
  //     for (auto &labelIdx : instr["labels"]) {
  //       std::cout << "STOI WITH LABEL IDX: " << labelIdx.get<std::string>() << std::endl;
  //       // int idx = std::stoi(labelIdx.get<std::string>());
  //       newLabels.push_back(labelIdx.get<std::string>());
  //     }
  //     instr["labels"] = newLabels;
  //   }
  // }
  
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

} 

namespace bril {

nlohmann::json mlirToJson(mlir::ModuleOp module) {
  MLIRToJSONConverter converter;
  return converter.convertModule(module);
}

} // namespace bril
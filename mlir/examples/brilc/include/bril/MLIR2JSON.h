#ifndef BRIL_MLIRTOJSON_H
#define BRIL_MLIRTOJSON_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "bril/Dialect.h"
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace bril {

/// Convert an MLIR module containing Bril dialect operations back to Bril JSON.
nlohmann::json mlirToJson(mlir::ModuleOp module);

} // namespace bril

#endif // BRIL_MLIRTOJSON_H
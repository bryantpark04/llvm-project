add_custom_target(Bril)
set_target_properties(Bril PROPERTIES FOLDER "MLIR/Examples")

macro(add_toy_chapter name)
  add_dependencies(Bril ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

# For a better template to copy, see examples/standalone
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_toy_chapter(brilc
  brilc.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/MLIR2JSON.cpp

  DEPENDS
  BrilOpsIncGen

  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(brilc
  PRIVATE
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRFuncDialect
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncTransforms
    MLIRDialectUtils
    MLIRTransforms)

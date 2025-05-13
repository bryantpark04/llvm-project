# Build

```
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DLLVM_CCACHE_BUILD=ON
cmake --build . --target check-mlir
```

# Translation

## Types
- int - IntegerType (I32)
- bool - IntegerType (I1)

## Instructions

In the future, these should probably be all custom Bril-dialect instructions that are then lowered into standard MLIR.

### Arithmetic
- add - arith::AddIOp
- mul - arith::MulIOp
- sub - arith::SubIOp
- div - arith::DivSIOp

### Comparison
All of these use arith::CmpIOp for integer comparison
- eq - arith::CmpIPredicate::eq
- lt - arith::CmpIPredicate::slt
- gt - arith::CmpIPredicate::sgt
- le - arith::CmpIPredicate::sle
- ge - arith::CmpIPredicate::sge

### Logic
- not - arith::XOrIOp
- and - arith::AndIOp
- or - arith::OrIOp

### Control
- jmp - cf::BranchOp
- br - cf::CondBranchOp
- call - func::CallOp
- ret - func::ReturnOp

### Miscellaneous
- id - bril::IDOp
- print - bril::PrintOp
- nop - bril::Nop
- label - MLIR Basic Block

### SSA
- phi (uses MLIR basic block arguments)

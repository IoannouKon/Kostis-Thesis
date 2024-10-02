#ifndef ROCC_CUSTOM_H
#define ROCC_CUSTOM_H

#include <stdint.h>

// Define custom function codes
#define CUSTOM_SET_ISBIAS 0
#define CUSTOM_LOAD_WEIGHTS_FUNCT 1
#define CUSTOM_LOAD_INPUTS_FUNCT 2
#define CUSTOM_READ_OUTPUT_FUNCT 3
#define CUSTOM_READ_BUSY_FUNCT 4  

#define STR1(x) #x
#define STR(x) STR1(x)
#define EXTRACT(a, size, offset) (((~(~0 << size) << offset) & a) >> offset)

#define CUSTOMX_OPCODE(x) CUSTOM_ ## x
#define CUSTOM_0 0b0001011
#define CUSTOM_1 0b0101011
#define CUSTOM_2 0b1011011
#define CUSTOM_3 0b1111011

#define CUSTOMX(X, xd, xs1, xs2, rd, rs1, rs2, funct) \
  CUSTOMX_OPCODE(X)                     |             \
  (rd                 << (7))           |             \
  (xs2                << (7+5))         |             \
  (xs1                << (7+5+1))       |             \
  (xd                 << (7+5+2))       |             \
  (rs1                << (7+5+3))       |             \
  (rs2                << (7+5+3+5))     |             \
  (EXTRACT(funct, 7, 0) << (7+5+3+5+5))

// Define ROCC flags
#define ROCC_XD  0x1  // Indicates write-back to rd
#define ROCC_XS1 0x2  // Indicates use of rs1
#define ROCC_XS2 0x4  // Indicates use of rs2

#define ROCC_INSTRUCTION_D(X, rd, funct) \
	ROCC_INSTRUCTION_R_I_I(X, rd, 0, 0, funct, 10) 

#define ROCC_INSTRUCTION_S(X, rs1, funct) \
	ROCC_INSTRUCTION_I_R_I(X, 0, rs1, 0, funct, 11)

#define ROCC_INSTRUCTION_S(X, rs1, funct) \
	ROCC_INSTRUCTION_I_R_I(X, 0, rs1, 0, funct, 11)    


#define ROCC_INSTRUCTION_R_I_I(X, rd, rs1, rs2, funct, rd_n) {           \
    register uint64_t rd_  asm ("x" # rd_n);                             \
    asm volatile (                                                       \
        ".word " STR(CUSTOMX(X, 1, 0, 0, rd_n, rs1, rs2, funct)) "\n\t"  \
        : "=r" (rd_));                                                   \
    rd = rd_;                                                            \
  }

#define ROCC_INSTRUCTION_I_R_I(X, rd, rs1, rs2, funct, rs1_n) {         \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 1, 0, rd, rs1_n, rs2, funct)) "\n\t" \
        :: [_rs1] "r" (rs1_));                                          \
  }
  
#define ROCC_INSTRUCTION_I_R_I(X, rd, rs1, rs2, funct, rs1_n) {         \
    register uint64_t rs1_ asm ("x" # rs1_n) = (uint64_t) rs1;          \
    asm volatile (                                                      \
        ".word " STR(CUSTOMX(X, 0, 1, 0, rd, rs1_n, rs2, funct)) "\n\t" \
        :: [_rs1] "r" (rs1_));                                          \
  }    


#endif // SRC_MAIN_C_ROCC_H
